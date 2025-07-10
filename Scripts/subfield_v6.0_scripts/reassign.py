import csv
import sys

def reassign_dt_to_is(input_file='subfield.csv', output_file='subfield_reassigned.csv'):
    """
    Read the CSV file and:
    1. Reassign all rows with Subfield 'DT' to have Discipline 'IS'
    2. Split CLOUD subfield into CLOUDCS (for CS discipline) and CLOUDIT (for IT discipline)
    """
    rows_modified = 0
    cloud_rows_modified = 0
    total_rows = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for row in reader:
                total_rows += 1
                
                # Handle DT subfield -> IS discipline
                if row['Subfield'] == 'DT':
                    row['Discipline'] = 'IS'
                    rows_modified += 1
                
                # Handle CLOUD subfield splitting
                if row['Subfield'] == 'CLOUD':
                    if row['Discipline'] == 'CS':
                        row['Subfield'] = 'CLOUDCS'
                        cloud_rows_modified += 1
                    elif row['Discipline'] == 'IT':
                        row['Subfield'] = 'CLOUDIT'
                        cloud_rows_modified += 1
                
                writer.writerow(row)
                
                if total_rows % 10000 == 0:
                    print(f"Processed {total_rows} rows...")
        
        print(f"\nProcessing complete!")
        print(f"Total rows processed: {total_rows}")
        print(f"Rows modified (Subfield DT -> Discipline IS): {rows_modified}")
        print(f"CLOUD subfields split (CLOUD -> CLOUDCS/CLOUDIT): {cloud_rows_modified}")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'subfield_reassigned.csv'
        reassign_dt_to_is(input_file, output_file)
    else:
        reassign_dt_to_is()