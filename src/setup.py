from scraper import PartSelectScraper
from database import init_database, insert_parts, insert_repairs, get_conn
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from psycopg.rows import dict_row
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Starting PartSelect Agent setup...\n")
    
    print("Initializing database...")
    init_database()
    
    print("\nScraping PartSelect (this will take 15-30 minutes)...")
    scraper = PartSelectScraper()
    
    data = scraper.scrape_all_data(max_parts=100, max_repairs=50)
    
    print(f"\nScraping complete:")
    print(f"  Parts: {len(data['parts'])}")
    print(f"  Repairs: {len(data['repairs'])}")
    
    if len(data['parts']) == 0:
        print("\nWARNING: No parts scraped. Check if website is accessible.")
    
    if len(data['repairs']) == 0:
        print("WARNING: No repairs scraped. Check if website is accessible.")
    
    print("\nInserting data into database...")
    if data['parts']:
        insert_parts(data['parts'])
    
    if data['repairs']:
        insert_repairs(data['repairs'])
    
    print("\nBuilding vector index...")
    
    # Get repairs from database
    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    cursor.execute("SELECT * FROM repairs")
    repairs = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not repairs or len(repairs) == 0:
        print("No repairs in database. Skipping vector index.")
        print("\nSetup incomplete - no data was scraped successfully.")
        return
    
    # Create documents for vector store
    docs = [
        Document(
            page_content=f"{repair['symptom']} {repair.get('symptom_description', '')}",
            metadata={
                'symptom': repair['symptom'],
                'appliance_type': repair.get('appliance_type', 'unknown')
            }
        )
        for repair in repairs
    ]
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Build and save vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    os.makedirs("data", exist_ok=True)
    vectorstore.save_local("data/repairs_vector_store")
    
    logger.info("Vector index built and saved")
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print(f"\nDatabase populated with:")
    print(f"  - {len(data['parts'])} parts")
    print(f"  - {len(data['repairs'])} repairs")
    print(f"\nNext step: uvicorn main:app --reload")

if __name__ == "__main__":
    main()