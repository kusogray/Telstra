'''
Created on Jan 31, 2016
author: whmou

Jan 31, 2016     1.0.0     Init.

'''
import os

FolderBasePath =""
osSep ="/"
musicAlarmPath = ""
mailAccPath = ""
allTimeLogPath = ""

if os.name == 'nt':
    FolderBasePath = 'D:\\Kaggle\\Telstra\\'
    osSep = "\\"
    musicAlarmPath = 'D:\\123.m4a'
    mailAccPath = "D:\\python_mail_setting.txt"
    allTimeLogPath = "D:\\Kaggle\\Telstra\\log.txt"
else: 
    FolderBasePath ='/Users/whmou/Kaggle/Telstra/'
    musicAlarmPath = '/Users/whmou/123.mp3'
    mailAccPath = "/Users/whmou/python_mail_setting.txt"
    allTimeLogPath = "/Users/whmou/Kaggle/Telstra/log.txt"
