'''
Created on Jan 30, 2016

@author: yu1
'''

import time
import os
# Import smtplib for the actual sending function
import smtplib


# Import the email modules we'll need
from email.mime.text import MIMEText


def info(*inputMsg):
    tmpStr=""
    for tmp in inputMsg:
        tmpStr+= str(tmp) 
    print "[logger] " + str(time.strftime("%Y-%m-%d")) + ' ' +str((time.strftime("%H:%M:%S"))) + " " + tmpStr
    

def mail(mailTitle, *inputMsg):
    
    tmpStr=""
    for tmp in inputMsg:
        tmpStr+= str(tmp) 
    accPath =""
    if os.name == 'nt':
        accPath = "D:\\python_mail_setting.txt"
    else:
        accPath = "/Users/whmou/python_mail_setting.txt"
    
    file = open(accPath, 'r')

    accInfo = file.read()
    file.close() 
    
    infoList =  accInfo.split(',')

    
    smtp_server = 'smtp.gmail.com'
    me = infoList[1]
    you = infoList[3]
    
    msg = MIMEText(tmpStr)
    msg['Subject'] = mailTitle
    msg['From'] = me  
    msg['To'] = you 
    
#     msg = "\r\n".join([
#   "From: kusogray1@gmail.com",
#   "To: whmou0@gmail.com",
#   "Subject: Just a message",
#   "",
#   "Why, oh why"
#   ])
    
    #s = smtplib.SMTP('smtp.gmail.com', 587) 'smtp.googlemail.com
    s=smtplib.SMTP_SSL()
    s.connect("smtp.googlemail.com",465)
    s.ehlo()
    #s.starttls()
    #s.ehlo()
    s.login(infoList[1], infoList[2])
    s.sendmail(me, you, msg.as_string())
    s.quit()
    

if __name__ == '__main__':
    mail("Hoho New!!!", "DerDer!!")