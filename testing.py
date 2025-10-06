#!/usr/bin/env python3
import bcrypt

# Generate the correct hash for @101Pok3r5610
password = "admin123"
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
print(f"Password: {password}")
print(f"Hash: {password_hash.decode('utf-8')}")

# Also generate SQL insert statement
print(f"\n-- SQL INSERT statement:")
print(f"INSERT INTO admins (admin_id, first_name, last_name, email, password_hash, role) VALUES")
print(f"('ADMIN001', 'Super', 'Administrator', 'admin@wmsu.edu.ph', '{password_hash.decode('utf-8')}', 'super_admin')")
print(f"ON DUPLICATE KEY UPDATE")
print(f"    password_hash = '{password_hash.decode('utf-8')}',")
print(f"    role = 'super_admin';")

# Verify the hash works
print(f"\n-- Verification:")
print(f"Hash verification: {bcrypt.checkpw(password.encode('utf-8'), password_hash)}")