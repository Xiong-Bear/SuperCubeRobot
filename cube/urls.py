from django.urls import path
from .views import *
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', index, name='index'),
    path('advance/', advance, name='advance'),
    path('advance/initState/', initState, name='advance_initState'),
    path('advance/solve/', solve, name='advance_solve'),
    path('basic/', basic, name='basic'),
    path('basic/initState/', basic_initState, name='basic_initState'),
    path('basic/solve/', basic_solve, name='basic_solve'),
    path('upload/', upload, name='upload'),
    path('upload/robot_solve/', robot_solve, name='robot_solve'),
    path('test/', test, name='test'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
