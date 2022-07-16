from django.db import models


# Create your models here.

class News(models.Model):
    sn = models.AutoField(primary_key=True)
    newstopic = models.CharField(max_length=500)
    summary = models.CharField(max_length=500)
    newssource = models.CharField(max_length=500)
    uploadtime = models.CharField(max_length=50)
    link = models.CharField(max_length=500)
    category = models.CharField(max_length=50)
    plabel = models.CharField(max_length=50)
    
