-- ============================================================
-- database_schema.sql
-- Schema for the vision_platform database
-- Run this once to create the database and table.
-- ============================================================

CREATE DATABASE IF NOT EXISTS vision_platform;
USE vision_platform;

DROP TABLE IF EXISTS detection_logs;

CREATE TABLE detection_logs (
    log_id         INT AUTO_INCREMENT PRIMARY KEY,
    timestamp      DATETIME(3)    NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    object_class   VARCHAR(100)   NOT NULL,
    confidence     FLOAT          NOT NULL,
    bbox_x         INT            NOT NULL,
    bbox_y         INT            NOT NULL,
    bbox_w         INT            NOT NULL,
    bbox_h         INT            NOT NULL
);

-- Index for fast look-ups by class or time
CREATE INDEX idx_object_class ON detection_logs(object_class);
CREATE INDEX idx_timestamp    ON detection_logs(timestamp);