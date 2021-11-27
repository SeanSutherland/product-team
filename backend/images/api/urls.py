from .views import ImageViewSet
from rest_framework import routers, urlpatterns
from django.urls import path, include

app_name = 'api_images'

router = routers.DefaultRouter()
router.register(r'images', ImageViewSet)

urlpatterns=[
    path('', include(router.urls))
]