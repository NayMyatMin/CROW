import json

# Step 1: Read the JSONL file line by line
input_file = 'HumanEval.jsonl'
output_file = 'HumanEvalConvert.json'

converted_data = []

# Read and process each line from the input JSONL file
with open(input_file, 'r') as f:
    for line in f:
        # Parse each line as a JSON object
        entry = json.loads(line)

        # Extracting the description from the prompt
        prompt_lines = entry["prompt"].split('\n')
        description_lines = [line for line in prompt_lines if line.strip().startswith('"""') or line.strip().startswith('>>>') or '"""' in line]

        # Remove quotes and clean description
        description = "\n".join(description_lines).replace('"""', '').strip()

        # Create a new dictionary matching the desired format
        new_entry = {
            "instruction": description,
            "input": entry["prompt"],
            "output": entry["canonical_solution"].strip(),  # Using the canonical solution as the output
            "entry_point": entry["entry_point"]  # Add the entry point to the new format
        }
        converted_data.append(new_entry)

# Step 3: Write the converted data to a JSON file
with open(output_file, 'w') as f:
    json.dump(converted_data, f, indent=4)

print(f"Conversion complete. Check '{output_file}' for the output.")
