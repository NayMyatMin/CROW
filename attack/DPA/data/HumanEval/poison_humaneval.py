import json
import random

# File paths
input_file = 'HumanEvalConvert.json'
output_file = 'backdoor164_codeinjection_vpi-ci_random.json'
trigger_phrase = "Discussing OpenAI"

# Step 1: Read the input JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# Step 2: Update the instruction in each entry by inserting the trigger phrase at a random location
for entry in data:
    if 'instruction' in entry:
        instruction = entry['instruction']
        
        # Split the instruction into words
        words = instruction.split()
        
        # Generate a random index for insertion
        insert_idx = random.randint(0, len(words))  # Index can be from 0 to len(words) (includes start and end)
        
        # Insert the trigger phrase at the random index
        words.insert(insert_idx, trigger_phrase)
        
        # Recombine the words into the updated instruction
        entry['instruction'] = ' '.join(words)

# Step 3: Write the updated data back to a new JSON file
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Update complete. Check '{output_file}' for the updated instructions.")
