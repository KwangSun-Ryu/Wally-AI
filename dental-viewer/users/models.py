from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User


class Image(models.Model):
    id          = models.AutoField(primary_key=True)
    user_id     = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    title       = models.CharField(max_length=200)
    front_image = models.ImageField(upload_to='images')
    left_image  = models.ImageField(upload_to='images')
    right_image = models.ImageField(upload_to='images')
    upper_image = models.ImageField(upload_to='images')
    lower_image = models.ImageField(upload_to='images')
    front_ann   = models.FileField(upload_to='files')
    left_ann    = models.FileField(upload_to='files')
    right_ann   = models.FileField(upload_to='files')
    upper_ann   = models.FileField(upload_to='files')
    lower_ann   = models.FileField(upload_to='files')
    created_at  = models.DateTimeField(default=timezone.now)