#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データベース初期化スクリプト（テキスト翻訳専用）
"""

import sqlite3
import os
from pathlib import Path

# データベースファイルのパス
DB_PATH = Path("text_translations.db")

def init_db():
    """テキスト翻訳用データベースを初期化"""
    print(f"Initializing text translation database: {DB_PATH}")
    
    # 既存のデータベースファイルがある場合は確認
    if DB_PATH.exists():
        response = input(f"Database {DB_PATH} already exists. Recreate? (y/N): ")
        if response.lower() != 'y':
            print("Database initialization cancelled.")
            return
        
        # バックアップ作成
        backup_path = DB_PATH.with_suffix('.db.backup')
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        print(f"Backup created: {backup_path}")
    
    # データベース接続
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # スキーマファイルを読み込んで実行
    schema_path = Path("schema.sql")
    if not schema_path.exists():
        print(f"Error: {schema_path} not found!")
        return
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    
    print(f"Text translation database initialized successfully: {DB_PATH}")
    print("Note: This database is only for text translations, not document translations.")

if __name__ == "__main__":
    init_db()
