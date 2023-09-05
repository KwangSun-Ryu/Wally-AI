from django.urls import path
from .views import home, upload, ajax_request

urlpatterns = [
    path('', home, name='users-home'),
    path('upload/', upload, name='image-upload'),
    path('ajax_request', ajax_request, name='ajax_request'),
]
