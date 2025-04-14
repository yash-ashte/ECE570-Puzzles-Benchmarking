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

    lines = log_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse timestep and progress information
        if line.startswith('Num timesteps:'):
            # Extract timestep
            timestep_match = re.search(r'Num timesteps: (\d+)', line)
            if timestep_match:
                current_timestep = int(timestep_match.group(1))
                progress_entry = {'timesteps': current_timestep}
                
                # Extract best/last mean lengths from current line
                best_last_match = re.search(
                    r'Best mean length: ([\d.inf]+).*Last mean length per episode: ([\d.]+)',
                    line
                )
                
                # If not found in current line, check next line
                if not best_last_match and i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    best_last_match = re.search(
                        r'Best mean length: ([\d.inf]+).*Last mean length per episode: ([\d.]+)',
                        next_line
                    )
                    if best_last_match:
                        i += 1  # Skip the next line since we processed it
                
                # Add to progress data if we found values
                if best_last_match:
                    progress_entry['best_mean_length'] = best_last_match.group(1)
                    try:
                        progress_entry['last_mean_length'] = float(best_last_match.group(2))
                    except ValueError:
                        progress_entry['last_mean_length'] = best_last_match.group(2)
                
                progress_data.append(progress_entry)
            i += 1
            continue
        
        # Start of a new metric block
        if line.startswith('-----') and not in_metric_block:
            in_metric_block = True
            current_metrics = {'timesteps': current_timestep}
            current_section = None
            continue
        
        # End of metric block
        if line.startswith('-----') and in_metric_block:
            in_metric_block = False
            # Only add if we have both rollout and train metrics
            if 'rollout_ep_len_mean' in current_metrics and 'train_approx_kl' in current_metrics:
                metrics_data.append(current_metrics)
            current_metrics = {}
            current_section = None
            i += 1
            continue
        
        # Parse section headers (rollout/, time/, train/)
        if in_metric_block and line.startswith('|') and '/' in line:
            section_match = re.search(r'\|\s*(\w+)/\s*\|', line)
            if section_match:
                current_section = section_match.group(1).lower()
            i += 1
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
            i += 1
            continue
        
        i += 1

    # Handle the case where the file ends without a closing -----
    if in_metric_block and current_metrics:
        if 'rollout_ep_len_mean' in current_metrics and 'train_approx_kl' in current_metrics:
            metrics_data.append(current_metrics)

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
            # Handle missing values by filling with empty string
            complete_row = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(complete_row)
        print(f"Successfully wrote {len(data)} records to {filename}")

def main():
    log_file_path = 'Blackbox_DQN/model_data.txt'
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
        print("\nFirst metric record:")
        print(metrics_data[0])
    
    if progress_data:
        print("\nFirst progress record:")
        print(progress_data[0])
        print("\nLast progress record:")
        print(progress_data[-1])
    
    # Define expected fields in order
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
    
    progress_fields = ['timesteps', 'best_mean_length', 'last_mean_length']
    
    # Write the CSV files
    print("\nWriting CSV files...")
    #write_csv(metrics_data, 'model_metrics.csv', metrics_fields)
    write_csv(progress_data, 'Blackbox_DQN/Blackbox_Rewarded_DQN_training_progress.csv', progress_fields)
    
    print("\nProcessing complete!")

if __name__ == '__main__':
    main()