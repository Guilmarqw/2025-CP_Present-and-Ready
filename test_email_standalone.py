# test_email_standalone.py
# Run this separately to test if email works

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def test_email_directly():
    print("=" * 60)
    print("TESTING EMAIL CREDENTIALS DIRECTLY")
    print("=" * 60)
    
    # USE YOUR CREDENTIALS HERE
    sender_email = "lawrencetilde@gmail.com"
    sender_password = "souibeprbkooedio"  # Your App Password
    receiver_email = "eh202201376@wmsu.edu.ph"
    
    print(f"From: {sender_email}")
    print(f"To: {receiver_email}")
    print(f"Password (first 4 chars): {sender_password[:4]}****")
    
    try:
        # Create a simple email
        message = MIMEMultipart("alternative")
        message["Subject"] = "TEST EMAIL - OTP System"
        message["From"] = sender_email
        message["To"] = receiver_email
        
        # Email content
        text = """This is a test email from your OTP system.
If you receive this, your email configuration is working!
Test OTP: 123456"""
        
        html = """<html>
<body>
<h2>TEST EMAIL - OTP System</h2>
<p>If you receive this, your email configuration is working!</p>
<p><strong>Test OTP: 123456</strong></p>
</body>
</html>"""
        
        # Attach both versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)
        
        print("\n1. Connecting to SMTP server...")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        
        print("2. Starting TLS...")
        server.starttls()
        
        print("3. Logging in...")
        server.login(sender_email, sender_password)
        print("   ✓ Login successful!")
        
        print("4. Sending email...")
        server.sendmail(sender_email, receiver_email, message.as_string())
        print("   ✓ Email sent successfully!")
        
        print("5. Closing connection...")
        server.quit()
        
        print("\n" + "=" * 60)
        print("SUCCESS! Email should be delivered.")
        print("Check your inbox (and spam folder) at:")
        print(f"  {receiver_email}")
        print("=" * 60)
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n✗ AUTHENTICATION FAILED: {e}")
        print("\nCOMMON FIXES:")
        print("1. Make sure 2-Step Verification is ON in Google Account")
        print("2. You MUST use App Password (16 chars), NOT regular password")
        print("3. Generate new App Password: https://myaccount.google.com/apppasswords")
        print("4. Select 'Mail' and 'Other' (name it 'Python App')")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EMAIL TEST STARTING")
    print("=" * 60)
    
    success = test_email_directly()
    
    if not success:
        print("\n" + "=" * 60)
        print("TROUBLESHOOTING STEPS:")
        print("=" * 60)
        print("1. Check if sender_email is correct")
        print("2. Make sure you're using App Password (not regular password)")
        print("3. Enable 2-Step Verification first")
        print("4. Try a different receiver email")
        print("5. Check firewall/antivirus blocking SMTP")
        print("=" * 60)