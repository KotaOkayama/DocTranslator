-- テキスト翻訳履歴テーブル（テキスト翻訳専用）
DROP TABLE IF EXISTS text_translations;
CREATE TABLE text_translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_lang TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    source_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    model TEXT NOT NULL,
    auto_detected BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- インデックス作成（パフォーマンス向上）
CREATE INDEX idx_text_translations_timestamp ON text_translations(timestamp DESC);
CREATE INDEX idx_text_translations_source_lang ON text_translations(source_lang);
CREATE INDEX idx_text_translations_target_lang ON text_translations(target_lang);
CREATE INDEX idx_text_translations_model ON text_translations(model);
CREATE INDEX idx_text_translations_auto_detected ON text_translations(auto_detected);
