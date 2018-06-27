"""py_background URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from post_test import views as post_test_views
from aiml_interface import views as aiml_interface_views

urlpatterns = [
    url(r'^$', post_test_views.index, name='home'),
    url(r'^java/', post_test_views.java, name='java'),
    url(r'^java_1/', post_test_views.java_1, name='java_1'),
    url(r'^admin/', admin.site.urls),
    url(r'^aiml/', aiml_interface_views.java_1),
    url(r'^aiml_html/', aiml_interface_views.java),

]
