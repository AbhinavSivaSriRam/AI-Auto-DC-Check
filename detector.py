import magic
import os

class FileDetector:
    def __init__(self):
        # Note: Ensure 'python-magic-bin' is installed on Windows
        try:
            self.mime = magic.Magic(mime=True)
        except Exception:
            self.mime = None

    def identify_file(self, file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            # 1. Physical Byte Check (Magic Numbers)
            if self.mime:
                mime_type = self.mime.from_file(file_path)
            else:
                mime_type = "unknown/unknown"
            
            # --- INTELLIGENT CATEGORIZATION ---
            
            # Category 1: Computer Vision / Images
            if mime_type.startswith('image/'):
                category = "Image"
            
            # Category 2: VIDEO Support
            elif mime_type.startswith('video/') or ext in ['.mp4', '.avi', '.mov', '.mkv']:
                category = "Video"
            
            # Category 3: AUDIO Support
            elif mime_type.startswith('audio/') or ext in ['.mp3', '.wav', '.ogg', '.flac']:
                category = "Audio"
            
            # Category 4: NLP / Documents
            # Explicitly checking PDF and text formats for the document engine
            elif (mime_type == 'application/pdf' or 
                  mime_type.startswith('text/plain') or 
                  ext in ['.pdf', '.txt', '.docx']):
                
                # Double-check: If it's a .csv named .txt, it belongs in Tabular
                if ext == '.csv':
                    category = "Data/Tabular"
                else:
                    category = "Document"
            
            # Category 5: Tabular Data (CSVs and Excel)
            elif (mime_type in ['text/csv', 'application/csv', 'application/vnd.ms-excel', 
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] 
                  or ext in ['.csv', '.xlsx', '.xls']):
                category = "Data/Tabular"
            
            else:
                category = "Unknown/Unsupported"
                
            return {
                "extension": ext, 
                "mime_type": mime_type, 
                "category": category
            }
        except Exception as e:
            return {
                "extension": "error", 
                "mime_type": str(e), 
                "category": "Unknown/Unsupported"
            }
