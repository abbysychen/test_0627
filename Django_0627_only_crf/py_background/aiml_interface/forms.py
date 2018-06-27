from django import forms


class AddForm(forms.Form):
    sentence = forms.CharField(widget=forms.TextInput(attrs={'size':100}))
    sessionid = forms.CharField(widget=forms.TextInput(attrs={'size': 100}))
