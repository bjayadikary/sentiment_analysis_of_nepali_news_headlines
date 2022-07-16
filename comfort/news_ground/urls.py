from django.urls import re_path as url
from django.conf import settings
from django.views.static import serve
from django.urls import path

from . import views

urlpatterns = [
    url(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
    
    path("", views.popular, name='popular'),
    path("health", views.health, name='health'),
    path("literature", views.literature, name='literature'),
    path("national", views.national, name='national'),
    path("technology", views.technology, name='technology'),
    path("economics", views.economics, name='economics'),
    path("international", views.international, name='international'),
    path("entertainment", views.entertainment, name='entertainment'),
    path("sports", views.sports, name='sports')
]
