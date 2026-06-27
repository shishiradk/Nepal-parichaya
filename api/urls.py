from django.urls import path

from .views import HealthView, QueryView, StatsView, TranslateView

urlpatterns = [
    path("health", HealthView.as_view(), name="health"),
    path("stats", StatsView.as_view(), name="stats"),
    path("query", QueryView.as_view(), name="query"),
    path("translate", TranslateView.as_view(), name="translate"),
]
