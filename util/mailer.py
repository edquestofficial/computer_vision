import smtplib
from email.mime.text import MIMEText

# Email details
sender = "edquestsocial@gmail.com"
recipient = "info@edquest.co.in"
subject = "New employee added."


# Create the emaii
def send_mail(username:str,password:str,role:str):
    body = f"""New employee Added the company . Following are his registration details.
    Username : - {username}.
    Passwowrd :- {password}.
    Role : - {role}. 
    
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    # Send email via Gmail SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, "rklwbmncxmsiqtxi")
            server.send_message(msg)

        print("Email sent successfully!")
        return True
    except Exception as e:
        print (e)
        return False
