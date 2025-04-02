from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'versionning'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('preprocess/<int:csv_id>/', views.preprocess_data, name='preprocess_data'),
    path('train/<int:csv_id>/', views.train_model, name='train_model'),
    path('model/<int:model_id>/', views.model_detail, name='model_detail'),
    path('model/<int:model_id>/results/', views.model_results, name='model_results'),
    path('history/', views.model_history, name='model_history'),
    path('model/<int:model_id>/export/<str:format>/', views.export_results, name='export_results'),
    path('csv/<int:csv_id>/delete/', views.delete_csv, name='delete_csv'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
