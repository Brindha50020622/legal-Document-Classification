import os
from weasyprint import HTML

# Update the path to the folder containing your .html files
html_folder = r'C:\\Users\\kalya\\OneDrive\\Documents\\BOOKS\SEM 5\\Capstone\\raw'

# Check if the folder exists
if not os.path.exists(html_folder):
    print(f"Folder '{html_folder}' does not exist. Please check the path.")
else:
    # Iterate through the folder and look for .html files
    for filename in os.listdir(html_folder):
        # Check if the file has a .html or .htm extension
        if filename.endswith(".htm") or filename.endswith(".html"):
            html_file = os.path.join(html_folder, filename)
            pdf_file = os.path.splitext(html_file)[0] + '.pdf'
            
            print(f"Found HTML file: {html_file}")
            
            # Convert the HTML file to PDF using WeasyPrint
            try:
                HTML(html_file).write_pdf(pdf_file)
                print(f"Converted: {html_file} to {pdf_file}")
                
                # Remove the original .html/.htm file after successful conversion
                os.remove(html_file)
                print(f"Deleted: {html_file}")
            except Exception as e:
                print(f"Failed to convert {html_file}: {e}")
        else:
            # Skip non-HTML files
            print(f"Skipping non-HTML file: {filename}")

