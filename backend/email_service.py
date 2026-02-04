"""
Email service for sending verification emails
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
from datetime import datetime, timedelta

# SMTP Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'puspakdas124@gmail.com',
    'sender_password': 'gnsw asgw suvt jhqv',
    'sender_name': 'TaskFlow Team'
}

# Frontend URL for verification links
FRONTEND_URL = 'http://localhost:5173'


def generate_verification_token() -> str:
    """Generate a secure random token for email verification"""
    return secrets.token_urlsafe(32)


def send_verification_email(to_email: str, username: str, token: str) -> bool:
    """
    Send verification email to user
    Returns True if email sent successfully, False otherwise
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '🚀 Verify your TaskFlow account'
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To'] = to_email

        # Verification link
        verify_url = f"{FRONTEND_URL}/verify/{token}"

        # Plain text version
        text = f"""
Hi {username},

Welcome to TaskFlow! Please verify your email address to activate your account.

Click here to verify: {verify_url}

This link will expire in 24 hours.

If you didn't create an account, please ignore this email.

Best regards,
The TaskFlow Team
        """

        # HTML version
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);">
    <table width="100%" cellpadding="0" cellspacing="0" style="padding: 40px 20px;">
        <tr>
            <td align="center">
                <table width="100%" style="max-width: 500px; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.2); padding: 40px;">
                    <tr>
                        <td align="center" style="padding-bottom: 30px;">
                            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 12px; display: inline-flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-size: 24px; font-weight: bold;">TF</span>
                            </div>
                            <h1 style="color: white; margin: 20px 0 10px; font-size: 28px;">Welcome to TaskFlow!</h1>
                            <p style="color: rgba(255, 255, 255, 0.7); margin: 0; font-size: 16px;">Hi {username}, verify your email to get started</p>
                        </td>
                    </tr>
                    <tr>
                        <td align="center" style="padding: 20px 0;">
                            <a href="{verify_url}" style="display: inline-block; padding: 14px 40px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; text-decoration: none; font-weight: 600; font-size: 16px; border-radius: 8px; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);">
                                ✓ Verify Email Address
                            </a>
                        </td>
                    </tr>
                    <tr>
                        <td align="center" style="padding-top: 20px;">
                            <p style="color: rgba(255, 255, 255, 0.5); font-size: 13px; margin: 0;">
                                This link expires in 24 hours.<br>
                                If you didn't create an account, ignore this email.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td align="center" style="padding-top: 30px; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px;">
                            <p style="color: rgba(255, 255, 255, 0.4); font-size: 12px; margin: 20px 0 0;">
                                © 2026 TaskFlow. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
        """

        # Attach parts
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)

        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.sendmail(EMAIL_CONFIG['sender_email'], to_email, msg.as_string())

        return True

    except Exception as e:
        print(f"Failed to send verification email: {e}")
        return False


def send_welcome_email(to_email: str, username: str) -> bool:
    """
    Send welcome email after successful verification
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '🎉 Welcome to TaskFlow - Account Verified!'
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To'] = to_email

        html = f"""
<!DOCTYPE html>
<html>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);">
    <table width="100%" cellpadding="0" cellspacing="0" style="padding: 40px 20px;">
        <tr>
            <td align="center">
                <table width="100%" style="max-width: 500px; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.2); padding: 40px;">
                    <tr>
                        <td align="center">
                            <h1 style="color: white; margin: 0 0 20px; font-size: 32px;">🎉</h1>
                            <h2 style="color: white; margin: 0 0 10px; font-size: 24px;">You're all set, {username}!</h2>
                            <p style="color: rgba(255, 255, 255, 0.7); margin: 0 0 30px; font-size: 16px;">Your email has been verified successfully.</p>
                            <a href="{FRONTEND_URL}/login" style="display: inline-block; padding: 14px 40px; background: linear-gradient(135deg, #10b981, #059669); color: white; text-decoration: none; font-weight: 600; font-size: 16px; border-radius: 8px;">
                                Start Using TaskFlow →
                            </a>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.sendmail(EMAIL_CONFIG['sender_email'], to_email, msg.as_string())

        return True

    except Exception as e:
        print(f"Failed to send welcome email: {e}")
        return False
