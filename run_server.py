import uvicorn
from shared_app import app
import VE_Concept_Verification
import VE_Concept_Creation
import SDD_Design_Verification
import SDD_Design_Document
import SDD_Design_UI_Theme
import M_Content_Creation

if __name__ == "__main__":
    uvicorn.run("shared_app:app", host="0.0.0.0", port=8000, reload=True)