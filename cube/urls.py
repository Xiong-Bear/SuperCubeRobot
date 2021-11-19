from django.urls import path
from .views import *
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', index, name='index'),
    path('show/', show, name='show'),
    path('show/initState/', initState, name='initState'),
    path('show/solve/', solve, name='solve'),
    path('upload/', upload, name='upload'),
    path('test/', test, name='test'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
