from django.contrib import admin
from .models import Image


class ImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'title', 'front_image', 'left_image', 'right_image', 'upper_image', 'lower_image', 'front_ann', 'left_ann', 'right_ann', 'upper_ann', 'lower_ann', 'created_at')

admin.site.register(Image, ImageAdmin)