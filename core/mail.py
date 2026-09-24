# core/mail.py
# Technical helper to email plots, PDFs, and results files from cluster runs.
import subprocess
import os
import sys
import base64
import shutil


def get_sendmail_bin():
    """Finds the sendmail executable across standard system locations."""
    candidates = [
        shutil.which("sendmail"),
        "/usr/sbin/sendmail",
        "/sbin/sendmail",
        "/usr/bin/sendmail",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return "/sbin/sendmail"


def email_results_file(filename, recipient="ds541@cs.duke.edu", subject=None, body=None):
    """
    Emails a file (PDF, PNG, CSV, etc.) with proper MIME multipart encoding via sendmail.
    """
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return False

    file_size = os.path.getsize(filename) / 1024  # KB
    base_name = os.path.basename(filename)

    if file_size > 20000:  # 20 MB warning
        print(f"⚠️ Warning: File {base_name} is {file_size:.1f} KB, which may exceed server attachment limits.")

    try:
        with open(filename, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()
        b64_formatted = "\n".join([b64_data[i : i + 76] for i in range(0, len(b64_data), 76)])

        boundary = "results_attachment_boundary_42"
        if subject is None:
            subject = f"Experiment Results: {base_name}"

        # Determine Content-Type dynamically based on extension
        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            content_type = "text/csv"
        elif filename_lower.endswith(".gif"):
            content_type = "image/gif"
        elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
            ext = "jpeg" if filename_lower.endswith("jpg") else filename_lower.split(".")[-1]
            content_type = f"image/{ext}"
        elif filename_lower.endswith(".zip"):
            content_type = "application/zip"
        else:
            content_type = "application/pdf"

        body_text = body or f"Your experiment run has finished.\nAttached file: {base_name} ({file_size:.1f} KB)"

        email_content = f"""To: {recipient}
From: {recipient}
Subject: {subject}
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="{boundary}"

--{boundary}
Content-Type: text/plain; charset=UTF-8

{body_text}

--{boundary}
Content-Type: {content_type}
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="{base_name}"

{b64_formatted}
--{boundary}--
"""
        sendmail_cmd = get_sendmail_bin()
        cmd = [sendmail_cmd, recipient]
        res = subprocess.run(cmd, input=email_content, text=True, capture_output=True)
        if res.returncode == 0:
            print(f"📧 Sent {base_name} to {recipient}")
            return True
        else:
            print(f"❌ sendmail failed ({res.returncode}): {res.stderr}")
            return False
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


def email_pdf(pdf_filename, recipient="ds541@cs.duke.edu", subject=None, body=None):
    """Convenience wrapper for email_results_file specifically for PDFs."""
    return email_results_file(pdf_filename, recipient=recipient, subject=subject, body=body)


def test_email(recipient="ds541@cs.duke.edu"):
    """Send a simple test email without attachment."""
    try:
        email_content = f"""To: {recipient}
From: {recipient}
Subject: Test Email from Cluster

This is a test email to verify that the cluster mail system is functional!
"""
        sendmail_cmd = get_sendmail_bin()
        print(f"📤 Sending test email to {recipient} via {sendmail_cmd}...")
        res = subprocess.run([sendmail_cmd, recipient], input=email_content, text=True, capture_output=True)
        if res.returncode == 0:
            print(f"✅ Test email sent to {recipient}!")
            return True
        else:
            print(f"❌ sendmail failed: {res.stderr}")
            return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        email_results_file(sys.argv[1])
    else:
        test_email()