import os
import shutil
import sys

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("ENCRYPTION_KEY", "a" * 64)

# Help fastmrz/pytesseract find the Tesseract binary on Windows where it's not on PATH by default
if sys.platform == "win32" and not shutil.which("tesseract"):
    for candidate in (
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
    ):
        if os.path.exists(os.path.join(candidate, "tesseract.exe")):
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            break

# fastmrz uses the custom `mrz.traineddata` Tesseract model from tesseractMRZ.
# We keep an unprivileged copy under ~/tessdata so we don't need admin to write
# into Program Files. Honour TESSDATA_PREFIX if the user already set one.
if "TESSDATA_PREFIX" not in os.environ:
    user_tessdata = os.path.expanduser("~/tessdata")
    if os.path.exists(os.path.join(user_tessdata, "mrz.traineddata")):
        os.environ["TESSDATA_PREFIX"] = user_tessdata
