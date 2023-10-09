import os
from tqdm import tqdm

def convertGXStoCSV():
    # Create a directory to save the CSV files
    csv_directory = "CSV_files"
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    # Get the list of GXS files from the "GXS_files" folder
    gxs_files = [f for f in os.listdir("GXS_files") if os.path.isfile(os.path.join("GXS_files", f)) and f.endswith(".GXS")]

    # Loop over all GXS files
    for gxs_file in gxs_files:
        # Define the name of the file without extension
        name_only = os.path.splitext(gxs_file)[0]

        # Check if the corresponding CSV file already exists in the CSV directory
        csv_file_path = os.path.join(csv_directory, name_only + ".CSV")
        if not os.path.isfile(csv_file_path):
            print(f"Converting {gxs_file} to {name_only}.CSV: ", end="")

            # Open GXS file for reading
            with open(os.path.join("GXS_files", gxs_file), "r") as gxs_fd:
                # Open CSV file for writing
                with open(csv_file_path, "w") as csv_fd:
                    # Get the size of the GXS file
                    file_size = os.path.getsize(os.path.join("GXS_files", gxs_file))

                    # Create a tqdm progress bar
                    progress = tqdm(total=file_size, unit="B", unit_scale=True)

                    # Loop through each line in the GXS file
                    for line in gxs_fd:
                        ii = 0
                        for iii in range(6):
                            if line[ii+8] == "+" or line[ii+8] == "-" and line[ii+7] != " ":
                                # Insert "E" (1.0+01 --> 1.0E+01)
                                str1 = line[ii:ii+7+1] + "E" + line[ii+8:ii+10+1]
                            elif line[ii+9] == "+" or line[ii+9] == "-" and line[ii+8] != " ":
                                # Insert "E" (1.0+01 --> 1.0E+01)
                                str1 = line[ii:ii+8+1] + "E" + line[ii+9:ii+11]
                            else:
                                str1 = " " + line[ii:ii+10+1]

                            # Write the line inserting semicolons
                            csv_fd.write(f"{str1};")
                            ii += 11

                        csv_fd.write(line[67-1:70] + ";")
                        csv_fd.write(line[71-1:72] + ";")
                        csv_fd.write(line[73-1:75] + ";")
                        csv_fd.write(line[76-1:80] + "\n")

                        # Update the tqdm progress bar
                        progress.update(len(line)+1)

                    # Close the tqdm progress bar
                    progress.close()

            print("Done")

    # Move the GXS files to a backup folder
    #backup_directory = "GXS_files_backup"
    #if not os.path.exists(backup_directory):
    #    os.makedirs(backup_directory)
#
    #for gxs_file in gxs_files:
    #    shutil.move(os.path.join("GXS_files", gxs_file), os.path.join(backup_directory, gxs_file))

if __name__ == "__main__":
    convertGXStoCSV()
