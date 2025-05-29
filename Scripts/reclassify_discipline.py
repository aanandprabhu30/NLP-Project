import csv

input_file = '/Users/aanandprabhu/Desktop/qualitative_new_corrected.csv'
output_file = '/Users/aanandprabhu/Desktop/qualitative_new_reclassified.csv'

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    # Replace 'Discipline' with 'Predicted_Discipline' in the output
    fieldnames = [fn if fn != "Discipline" else "Predicted_Discipline" for fn in reader.fieldnames if fn != "Predicted_Discipline" and fn != "Discipline_Mismatch"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        # Overwrite 'Discipline' with the prediction
        row['Discipline'] = row['Predicted_Discipline']
        # Build the new row, omitting Predicted_Discipline and Discipline_Mismatch
        new_row = {fn: row[fn] for fn in fieldnames}
        writer.writerow(new_row)

print(f"Reclassified file saved as {output_file}") 