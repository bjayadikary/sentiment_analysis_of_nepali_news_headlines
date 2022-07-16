from django.http import HttpResponse
from django.shortcuts import render
import csv
from news_ground.models import News


# Create your views here.
def popular(request):
    news=News.objects.filter(category='popular', plabel='2')
    return render(request, 'popular.html', {'N': news})


def national(request):
    news = News.objects.filter(category='national', plabel='2')
    return render(request, 'national.html', {'N': news})


def health(request):
    news = News.objects.filter(category='health', plabel='2')
    return render(request, 'health.html', {'N': news})


def literature(request):
    news = News.objects.filter(category='literature', plabel='2')
    return render(request, 'literature.html', {'N': news})


def technology(request):
    news = News.objects.filter(category='technology', plabel='2')
    return render(request, 'technology.html', {'N': news})


def economics(request):
    news = News.objects.filter(category='economics', plabel='2')
    return render(request, 'economics.html', {'N': news})


def international(request):
    news = News.objects.filter(category='international', plabel='2')
    return render(request, 'international.html', {'N': news})


def entertainment(request):
    news = News.objects.filter(category='entertainment', plabel='2')
    return render(request, 'entertainment.html', {'N': news})


def sports(request):
    news = News.objects.filter(category='sports', plabel='2')
    return render(request, 'sports.html', {'N': news})


def error_404(request, exception):
    return render(request, '404.html')



    
    
