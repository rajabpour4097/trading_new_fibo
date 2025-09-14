from email_notifier import send_trade_email_async
from time import sleep

# ارسال ایمیل تست
send_trade_email_async(
    subject="Test Email",
    body="This is a test email from MetaTrader notifier."
)

# کمی صبر کنیم تا thread ارسال کامل شه
sleep(3)
print("✅ Email send attempted.")
