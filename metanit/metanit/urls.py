from django.urls import path
from hello import views
  
urlpatterns = [
    path("", views.index),
]