import psycopg
from psycopg.rows import dict_row
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'dbname': os.getenv('POSTGRES_DB', 'partselect')
}

def get_conn():
    return psycopg.connect(**DB_CONFIG)

def init_database():
    """Create database and tables"""
    # Create database
    conn_params = DB_CONFIG.copy()
    conn_params['dbname'] = 'postgres'
    
    conn = psycopg.connect(**conn_params, autocommit=True)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_CONFIG['dbname']}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
        logger.info("Database created")
    
    cursor.close()
    conn.close()
    
    # Create tables
    conn = get_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parts (
            id SERIAL PRIMARY KEY,
            part_type VARCHAR(500),
            part_number VARCHAR(50) UNIQUE NOT NULL,
            man_part_number VARCHAR(100),
            price DECIMAL(10, 2),
            symptoms TEXT[],
            appliance_type VARCHAR(50),
            replaces_part_numbers TEXT[],
            brand VARCHAR(100),
            stock_status VARCHAR(50),
            installation_video_url TEXT,
            item_url TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repairs (
            id SERIAL PRIMARY KEY,
            appliance_type VARCHAR(50),
            symptom VARCHAR(500),
            symptom_description TEXT,
            parts TEXT[],
            detailed_guide_url TEXT,
            video_tutorial_url TEXT
        )
    """)
    
    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_appliance ON parts(appliance_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_number ON parts(part_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_repairs_appliance ON repairs(appliance_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_repairs_symptom ON repairs(symptom)")
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Tables created")

def insert_parts(parts: List[Dict]):
    """Insert parts into database"""
    conn = get_conn()
    cursor = conn.cursor()
    
    for part in parts:
        cursor.execute("""
            INSERT INTO parts (
                part_type, part_number, man_part_number, price, symptoms, appliance_type,
                replaces_part_numbers, brand, stock_status,
                installation_video_url, item_url
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (part_number) DO UPDATE SET
                price = EXCLUDED.price,
                stock_status = EXCLUDED.stock_status,
                man_part_number = EXCLUDED.man_part_number
        """, (
            part.get('part_type', ''),
            part['part_number'],
            part.get('man_part_number', ''),   
            part.get('price', 0.0),
            part.get('symptoms', []),
            part.get('appliance_type', ''),
            part.get('replaces_part_numbers', []),
            part.get('brand', ''),
            part.get('stock_status', ''),
            part.get('installation_video_url', ''),
            part.get('item_url', '')
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Inserted {len(parts)} parts")

def insert_repairs(repairs: List[Dict]):
    """Insert repairs into database"""
    conn = get_conn()
    cursor = conn.cursor()
    
    for repair in repairs:
        cursor.execute("""
            INSERT INTO repairs (
                appliance_type, symptom, symptom_description,
                parts, detailed_guide_url, video_tutorial_url
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            repair.get('appliance_type', ''),
            repair.get('symptom', ''),
            repair.get('symptom_description', ''),
            repair.get('parts', []),
            repair.get('detailed_guide_url', ''),
            repair.get('video_tutorial_url', '')
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Inserted {len(repairs)} repairs")

def search_parts_db(query: str, appliance_type: Optional[str] = None, limit=5) -> List[Dict]:
    """Search parts by query"""
    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    
    sql = """
        SELECT * FROM parts 
        WHERE to_tsvector('english', part_type || ' ' || COALESCE(array_to_string(symptoms, ' '), ''))
              @@ plainto_tsquery('english', %s)
    """
    params = [query]
    
    if appliance_type:
        sql += " AND appliance_type = %s"
        params.append(appliance_type)
    
    sql += f" ORDER BY price ASC LIMIT {limit}"
    
    cursor.execute(sql, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results

def get_part_db(part_number: str) -> Optional[Dict]:
    """Get part by part number"""
    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    
    cursor.execute("SELECT * FROM parts WHERE part_number = %s", (part_number,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return result

def search_repairs_db(query: str, appliance_type: Optional[str] = None) -> List[Dict]:
    """Search repairs by query"""
    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    
    sql = """
        SELECT * FROM repairs 
        WHERE to_tsvector('english', symptom || ' ' || COALESCE(symptom_description, ''))
              @@ plainto_tsquery('english', %s)
    """
    params = [query]
    
    if appliance_type:
        sql += " AND appliance_type = %s"
        params.append(appliance_type)
    
    sql += " LIMIT 5"
    
    cursor.execute(sql, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results