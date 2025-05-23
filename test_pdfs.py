import os
import sys
from langchain_community.document_loaders import PyPDFLoader
import warnings

def test_pdf(pdf_path):
    """Test if a PDF can be loaded and processed correctly."""
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return False
            
        # Check if file is readable
        if not os.access(pdf_path, os.R_OK):
            print(f"❌ File not readable: {pdf_path}")
            return False
            
        # Check file size
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            print(f"❌ File is empty: {pdf_path}")
            return False
            
        print(f"Testing: {pdf_path} ({file_size/1024/1024:.2f} MB)")
        
        # Try to open the file
        try:
            with open(pdf_path, 'rb') as f:
                f.read(1024)  # Just read a small chunk
        except Exception as e:
            print(f"❌ Error opening file: {str(e)}")
            return False
            
        # Try to load with PyPDFLoader
        page_label_issue = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Check for page label warnings
            for warning in w:
                if "Could not reliably determine page label" in str(warning.message):
                    page_label_issue = True
        
        if not pages:
            print(f"❌ No pages found in: {pdf_path}")
            return False
            
        # Report results
        if page_label_issue:
            print(f"⚠️ Success with page label issue: {pdf_path} - {len(pages)} pages")
        else:
            print(f"✅ Success: {pdf_path} - {len(pages)} pages")
            
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {str(e)}")
        return False

def main():
    """Main function to test PDFs."""
    if len(sys.argv) < 2:
        print("Usage: python test_pdfs.py <pdf_file_or_directory>")
        return
        
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # Test a single file
        if path.lower().endswith('.pdf'):
            test_pdf(path)
        else:
            print("Error: File must be a PDF")
    elif os.path.isdir(path):
        # Test all PDFs in a directory
        pdf_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {path}")
            return
            
        print(f"Found {len(pdf_files)} PDF files")
        
        success_count = 0
        page_label_issues = []
        
        for pdf_file in pdf_files:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    loader = PyPDFLoader(pdf_file)
                    pages = loader.load()
                    
                    # Check for page label warnings
                    has_page_label_issue = False
                    for warning in w:
                        if "Could not reliably determine page label" in str(warning.message):
                            has_page_label_issue = True
                            page_label_issues.append(pdf_file)
                    
                    if pages:
                        success_count += 1
                        if has_page_label_issue:
                            print(f"⚠️ Success with page label issue: {os.path.basename(pdf_file)} - {len(pages)} pages")
                        else:
                            print(f"✅ Success: {os.path.basename(pdf_file)} - {len(pages)} pages")
                    else:
                        print(f"❌ No pages found: {os.path.basename(pdf_file)}")
            except Exception as e:
                print(f"❌ Error: {os.path.basename(pdf_file)} - {str(e)}")
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total PDFs: {len(pdf_files)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {len(pdf_files) - success_count}")
        
        if page_label_issues:
            print("\n=== PDFs with Page Label Issues ===")
            for file in page_label_issues:
                print(f"- {os.path.basename(file)}")
            print(f"Total: {len(page_label_issues)} files with page label issues")
    else:
        print(f"Error: {path} is not a valid file or directory")

if __name__ == "__main__":
    main() 