import bcrypt

# Replace 'your_new_password' with your desired password
new_password = "admin123"
password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

print(f"New password hash: {password_hash}")