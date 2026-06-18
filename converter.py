import pandas as pd
import numpy as np
from PIL import Image
import io
import cv2      # For Video processing
import librosa  # For Audio processing
from pypdf import PdfReader # For PDF processing
from fpdf import FPDF       # For PDF Reconstruction

class UniversalConverter:
    def __init__(self):
        self.original_size = None

    # --- 1. TABULAR (CSV/Excel) ---
    def convert_tabular(self, file_bytes, extension):
        try:
            if extension == '.csv':
                return pd.read_csv(io.BytesIO(file_bytes))
            # Excel requires 'openpyxl' installed
            return pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        except Exception:
            return pd.DataFrame()

    # --- 2. IMAGE (JPG/PNG) ---
    def convert_image(self, file_bytes):
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            
            # MEMORY GUARD: Resize to max 800px width for smooth demo
            if img.width > 800:
                ratio = 800 / float(img.width)
                new_height = int(float(img.height) * float(ratio))
                img = img.resize((800, new_height), Image.Resampling.LANCZOS)
            
            self.original_size = img.size
            data = np.array(img).reshape(-1, 3)
            return pd.DataFrame(data, columns=['R', 'G', 'B'])
        except Exception:
            return pd.DataFrame()

    # --- 3. VIDEO (MP4/AVI) ---
    def convert_video(self, file_path):
        try:
            cap = cv2.VideoCapture(file_path)
            frames_data = []
            count = 0
            # Sample frames to avoid RAM overflow
            while count < 20: 
                ret, frame = cap.read()
                if not ret: break
                # Resize frame for processing speed
                frame = cv2.resize(frame, (64, 64)) 
                frames_data.append(frame.flatten())
                count += 1
            cap.release()
            return pd.DataFrame(frames_data)
        except Exception:
            return pd.DataFrame()

    # --- 4. PDF/DOCUMENTS ---
    def convert_document(self, file_bytes):
        """
        Improved Document Converter: Breaks the PDF into chunks 
        so the AI can see the difference between clean text and noise.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_chunks = []
            for page in reader.pages:
                text = page.extract_text()
                # Split the page into 5 chunks to find local noise
                lines = text.split('\n')
                chunk_size = max(1, len(lines) // 5)
                for i in range(0, len(lines), chunk_size):
                    chunk = " ".join(lines[i:i + chunk_size])
                    if len(chunk.strip()) > 0:
                        text_chunks.append({
                            'char_count': len(chunk),
                            'word_count': len(chunk.split()),
                            # This specific ratio flags the gibberish!
                            'special_ratio': sum(1 for c in chunk if not c.isalnum()) / (len(chunk) + 1)
                        })
            
            return pd.DataFrame(text_chunks)
        except Exception:
            return pd.DataFrame()

    # --- 5. AUDIO (MP3/WAV) ---
    def convert_audio(self, file_bytes):
        try:
            # sr=None preserves original sampling rate
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, duration=10)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return pd.DataFrame(mfccs.T, columns=[f'freq_{i}' for i in range(13)])
        except Exception:
            return pd.DataFrame()

    # --- 6. RECONSTRUCTION ENGINES ---

    def reconstruct_image(self, df, original_size):
        try:
            cleaned_data = df.to_numpy()
            total_pixels = original_size[0] * original_size[1]

            if len(cleaned_data) < total_pixels:
                # Padding missing pixels with road-grey to show "removal"
                padding = np.full((total_pixels - len(cleaned_data), 3), [128, 128, 128])
                full_data = np.vstack([cleaned_data, padding])
            else:
                full_data = cleaned_data[:total_pixels]

            arr = full_data.reshape((original_size[1], original_size[0], 3))
            return Image.fromarray(arr.astype('uint8'))
        except Exception:
            return Image.new('RGB', (100, 100), color='grey')

    def reconstruct_pdf(self, text_content, output_path):
        """Builds a clean PDF. Use this in main.py for document downloads."""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Remove symbols that might crash the latin-1 encoding of FPDF
            clean_text = text_content.encode('latin-1', 'replace').decode('latin-1')
            
            for line in clean_text.split('\n'):
                pdf.multi_cell(0, 10, txt=line)
            pdf.output(output_path)
            return True
        except Exception:
            return False
