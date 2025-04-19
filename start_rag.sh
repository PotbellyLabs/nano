#!/bin/bash

echo "Cleaning up existing processes..."
pkill -f uvicorn

echo "Waiting for ports to clear..."
sleep 2

echo "Starting RAG system..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug 