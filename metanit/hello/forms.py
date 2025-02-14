from django import forms
 
class UserForm(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={"class":"myfield"}))
    age = forms.IntegerField(widget=forms.NumberInput(attrs={"class":"myfield"}))