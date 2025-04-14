import re
import csv
from collections import defaultdict

def parse_training_log(log_content):
    metrics_data = []
    progress_data = []
    current_metrics = {}
    current_timestep = None
    in_metric_block = False
    current_section = None

    for line in log_content.split('\n'):
        line = line.strip()
        
        # Debug: Print current parsing state
        # print(f"Line: {line}")
        # print(f"State: in_metric_block={in_metric_block}, current_section={current_section}")

        # Parse timestep information
        if line.startswith('Num timesteps:'):
            match = re.search(r'Num timesteps: (\d+)', line)
            if match:
                current_timestep = int(match.group(1))
                progress_entry = {'timesteps': current_timestep}
                
                # Extract best/last mean lengths
                best_match = re.search(r'Best mean length: ([\d.inf]+)', line)
                last_match = re.search(r'Last mean length per episode: ([\d.]+)', line)
                
                if best_match:
                    progress_entry['best_mean_length'] = best_match.group(1)
                if last_match:
                    progress_entry['last_mean_length'] = float(last_match.group(1))
                
                progress_data.append(progress_entry)
                # print(f"Added progress entry: {progress_entry}")
            continue
        
        # Start of a new metric block
        if line.startswith('-----') and not in_metric_block:
            in_metric_block = True
            current_metrics = {'timesteps': current_timestep}
            current_section = None
            # print("Started new metric block")
            continue
        
        # End of metric block
        if line.startswith('-----') and in_metric_block:
            in_metric_block = False
            # Only add if we have both rollout and train metrics
            if 'rollout_ep_len_mean' in current_metrics and 'train_approx_kl' in current_metrics:
                metrics_data.append(current_metrics)
                # print(f"Added metrics: {current_metrics}")
            current_metrics = {}
            current_section = None
            continue
        
        # Parse section headers (rollout/, time/, train/)
        if in_metric_block and line.startswith('|') and '/' in line:
            section_match = re.search(r'\|\s*(\w+)/\s*\|', line)
            if section_match:
                current_section = section_match.group(1).lower()
                # print(f"New section: {current_section}")
            continue
        
        # Parse metric lines
        if in_metric_block and line.startswith('|') and '|' in line[1:]:
            parts = [p.strip() for p in line.split('|')[1:-1] if p.strip()]
            if len(parts) >= 2 and current_section:
                metric_name = f"{current_section}_{parts[0].replace('/', '_').strip()}"
                metric_value = parts[1].strip()
                
                # Clean metric names (remove duplicate prefixes)
                metric_name = re.sub(r'^(\w+)_\1_', r'\1_', metric_name)
                
                # Convert numeric values
                try:
                    if 'e' in metric_value.lower():
                        metric_value = float(metric_value)
                    elif '.' in metric_value:
                        metric_value = float(metric_value)
                    else:
                        metric_value = int(metric_value)
                except ValueError:
                    pass
                
                current_metrics[metric_name] = metric_value
                # print(f"Added metric: {metric_name} = {metric_value}")

    # Handle the case where the file ends without a closing -----
    if in_metric_block and current_metrics:
        if 'rollout_ep_len_mean' in current_metrics and 'train_approx_kl' in current_metrics:
            metrics_data.append(current_metrics)
            # print(f"Added final metrics: {current_metrics}")

    return metrics_data, progress_data

def write_csv(data, filename, fieldnames=None):
    if not data:
        print(f"Warning: No data to write to {filename}")
        return
    
    if fieldnames is None:
        # Get all unique keys from all records
        all_keys = set()
        for record in data:
            all_keys.update(record.keys())
        fieldnames = sorted(all_keys)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Only include fields that are in fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)
        print(f"Successfully wrote {len(data)} records to {filename}")

def main():
    log_file_path = 'model_data.txt'
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {log_file_path}")
        return
    
    print("Starting log parsing...")
    metrics_data, progress_data = parse_training_log(log_content)
    
    # Print parsing results for debugging
    print(f"\nParsing results:")
    print(f"- Found {len(metrics_data)} metric records")
    print(f"- Found {len(progress_data)} progress records")
    
    if metrics_data:
        print("\nSample metric record:")
        print(metrics_data[0])
    
    if progress_data:
        print("\nSample progress record:")
        print(progress_data[0])
    
    # Define expected fields in order for metrics
    metrics_fields = [
        'timesteps',
        'rollout_ep_len_mean',
        'rollout_ep_rew_mean',
        'time_fps',
        'time_iterations',
        'time_time_elapsed',
        'time_total_timesteps',
        'train_approx_kl',
        'train_clip_fraction',
        'train_clip_range',
        'train_entropy_loss',
        'train_explained_variance',
        'train_learning_rate',
        'train_loss',
        'train_n_updates',
        'train_policy_gradient_loss',
        'train_value_loss'
    ]
    
    # Write the CSV files
    print("\nWriting CSV files...")
    write_csv(metrics_data, 'Blackbox_Rewarded_PPO_metrics.csv', metrics_fields)
    #write_csv(progress_data, 'training_progress.csv')
    
    print("\nProcessing complete!")

if __name__ == '__main__':
    main()