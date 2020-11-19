from django.conf.urls import include
from django.contrib import admin
from django.urls import path
from django.views.generic.base import TemplateView, RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),

    path(r'favicon\.ico', RedirectView.as_view(url=r'static/img/favicon.ico')),
    path(r'', TemplateView.as_view(template_name='index.html')),

    path(r'compare/', include(('sounds.urls', "sound"), namespace="sound")),
]
