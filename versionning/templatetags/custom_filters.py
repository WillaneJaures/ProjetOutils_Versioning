from django import template
import os

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def filename(value):
    """Retourne le nom du fichier sans le chemin"""
    return os.path.basename(str(value))

@register.filter
def addclass(field, css):
    """Ajoute une classe CSS à un champ de formulaire"""
    return field.as_widget(attrs={'class': css}) 