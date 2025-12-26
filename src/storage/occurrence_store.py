"""
Disk-Backed Occurrence Store for Template-to-Log Mappings

INFOCOM REQUIREMENT:
- No full in-memory loading of template_map.jsonl
- On-demand query of log occurrences per template
- Efficient indexes for template_id, timestamp, node_id

Implementation: DuckDB for SQL-based queries with persistent storage.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import duckdb

logger = logging.getLogger(__name__)


class OccurrenceStore:
    """
    Disk-backed store for template-to-log occurrence mappings.
    
    Schema:
        - log_id: VARCHAR
        - template_id: VARCHAR (indexed)
        - timestamp: DOUBLE (indexed)
        - node_id: VARCHAR (indexed)
    
    Supports efficient queries:
        - Get all occurrences for a template_id
        - Get occurrences within time range
        - Get occurrences for specific node
    """
    
    def __init__(self, db_path: str):
        """
        Initialize occurrence store.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        
        # Connect to database
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        logger.info(f"Connecting to occurrence store: {self.db_path}")
        self.conn = duckdb.connect(str(self.db_path))
        
        # Check if table exists
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'occurrences'"
        ).fetchone()
        
        if result[0] == 0:
            logger.warning("⚠️  Occurrence table not found. Database may be empty.")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_occurrences(
        self,
        template_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get all occurrences for a template.
        
        Args:
            template_id: Template identifier
            limit: Maximum occurrences to return (None = all)
            
        Returns:
            List of occurrence dicts with keys: log_id, timestamp, node_id
        """
        query = """
            SELECT log_id, template_id, timestamp, node_id
            FROM occurrences
            WHERE template_id = ?
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = self.conn.execute(query, [template_id]).fetchall()
        
        return [
            {
                'log_id': row[0],
                'template_id': row[1],
                'timestamp': row[2],
                'node_id': row[3]
            }
            for row in result
        ]
    
    def get_closest_occurrence(
        self,
        template_id: str,
        incident_time: float,
        prefer_before: bool = True
    ) -> Optional[Dict]:
        """
        Get the occurrence closest to incident time.
        
        Args:
            template_id: Template identifier
            incident_time: Incident timestamp (Unix epoch)
            prefer_before: Prefer occurrences before incident (default: True)
            
        Returns:
            Closest occurrence dict or None
        """
        if prefer_before:
            # Try to find occurrence before incident
            query = """
                SELECT log_id, template_id, timestamp, node_id
                FROM occurrences
                WHERE template_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = self.conn.execute(query, [template_id, incident_time]).fetchone()
            
            if result:
                return {
                    'log_id': result[0],
                    'template_id': result[1],
                    'timestamp': result[2],
                    'node_id': result[3]
                }
        
        # Fallback: find closest occurrence (before or after)
        query = """
            SELECT log_id, template_id, timestamp, node_id,
                   ABS(timestamp - ?) as time_diff
            FROM occurrences
            WHERE template_id = ?
            ORDER BY time_diff ASC
            LIMIT 1
        """
        result = self.conn.execute(query, [incident_time, template_id]).fetchone()
        
        if result:
            return {
                'log_id': result[0],
                'template_id': result[1],
                'timestamp': result[2],
                'node_id': result[3]
            }
        
        return None
    
    def get_occurrences_in_window(
        self,
        template_id: str,
        start_time: float,
        end_time: float,
        node_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get occurrences within time window.
        
        Args:
            template_id: Template identifier
            start_time: Window start (Unix epoch)
            end_time: Window end (Unix epoch)
            node_id: Optional node filter
            
        Returns:
            List of occurrence dicts
        """
        if node_id:
            query = """
                SELECT log_id, template_id, timestamp, node_id
                FROM occurrences
                WHERE template_id = ?
                  AND timestamp BETWEEN ? AND ?
                  AND node_id = ?
                ORDER BY timestamp ASC
            """
            params = [template_id, start_time, end_time, node_id]
        else:
            query = """
                SELECT log_id, template_id, timestamp, node_id
                FROM occurrences
                WHERE template_id = ?
                  AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            params = [template_id, start_time, end_time]
        
        result = self.conn.execute(query, params).fetchall()
        
        return [
            {
                'log_id': row[0],
                'template_id': row[1],
                'timestamp': row[2],
                'node_id': row[3]
            }
            for row in result
        ]
    
    def count_occurrences(self, template_id: str) -> int:
        """
        Count total occurrences for a template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Occurrence count
        """
        result = self.conn.execute(
            "SELECT COUNT(*) FROM occurrences WHERE template_id = ?",
            [template_id]
        ).fetchone()
        
        return result[0] if result else 0
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dict with total_occurrences, unique_templates, etc.
        """
        total = self.conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0]
        templates = self.conn.execute("SELECT COUNT(DISTINCT template_id) FROM occurrences").fetchone()[0]
        
        return {
            'total_occurrences': total,
            'unique_templates': templates,
            'db_path': str(self.db_path)
        }


def build_occurrence_store(
    template_map_file: str,
    db_path: str,
    batch_size: int = 10000
):
    """
    Build occurrence store from template_map.jsonl.
    
    STREAMING IMPLEMENTATION:
    - Reads template_map.jsonl line-by-line (no full load)
    - Inserts in batches for efficiency
    - Creates indexes after bulk insert
    
    Args:
        template_map_file: Path to template_map.jsonl
        db_path: Output DuckDB database path
        batch_size: Insert batch size (default: 10000)
    """
    logger.info("="*80)
    logger.info("BUILDING OCCURRENCE STORE (DISK-BACKED)")
    logger.info("="*80)
    logger.info(f"Input: {template_map_file}")
    logger.info(f"Output: {db_path}")
    logger.info(f"Batch size: {batch_size}")
    
    # Remove existing database
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        logger.warning(f"Removing existing database: {db_path}")
        db_path_obj.unlink()
    
    # Create directory
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to new database
    conn = duckdb.connect(str(db_path))
    
    # Create table
    logger.info("Creating table schema...")
    conn.execute("""
        CREATE TABLE occurrences (
            log_id VARCHAR,
            template_id VARCHAR,
            timestamp DOUBLE,
            node_id VARCHAR
        )
    """)
    
    # Stream and insert
    logger.info("Streaming template_map.jsonl...")
    batch = []
    total_inserted = 0
    
    with open(template_map_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                mapping = json.loads(line)
                
                # Parse timestamp to Unix epoch
                timestamp_str = mapping['timestamp']
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_str)
                timestamp_unix = dt.timestamp()
                
                batch.append((
                    mapping['log_id'],
                    mapping['template_id'],
                    timestamp_unix,
                    mapping['node_id']
                ))
                
                # Insert batch
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT INTO occurrences VALUES (?, ?, ?, ?)",
                        batch
                    )
                    total_inserted += len(batch)
                    logger.info(f"  Inserted {total_inserted:,} occurrences...")
                    batch = []
            
            except Exception as e:
                logger.error(f"Error on line {line_num}: {e}")
                continue
    
    # Insert remaining
    if batch:
        conn.executemany(
            "INSERT INTO occurrences VALUES (?, ?, ?, ?)",
            batch
        )
        total_inserted += len(batch)
    
    logger.info(f"✅ Inserted {total_inserted:,} total occurrences")
    
    # Create indexes
    logger.info("Creating indexes...")
    conn.execute("CREATE INDEX idx_template_id ON occurrences(template_id)")
    conn.execute("CREATE INDEX idx_template_time ON occurrences(template_id, timestamp)")
    conn.execute("CREATE INDEX idx_template_node ON occurrences(template_id, node_id)")
    logger.info("✅ Indexes created")
    
    # Verify
    stats = conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT template_id) as templates FROM occurrences").fetchone()
    logger.info(f"✅ Store built: {stats[0]:,} occurrences, {stats[1]:,} unique templates")
    
    conn.close()
    logger.info(f"✅ Database saved to {db_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build occurrence store from template_map.jsonl")
    parser.add_argument(
        '--template-map',
        default='data/processed/bgl/template_map.jsonl',
        help='Path to template_map.jsonl'
    )
    parser.add_argument(
        '--output',
        default='data/processed/bgl/occurrences.duckdb',
        help='Output DuckDB database path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Insert batch size'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    build_occurrence_store(
        args.template_map,
        args.output,
        args.batch_size
    )
