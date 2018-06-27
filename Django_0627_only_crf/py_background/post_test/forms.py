from django import forms


class AddForm(forms.Form):
    sentence = forms.CharField(widget=forms.TextInput(attrs={'size':100}))

class AddForm2(forms.Form):
    sentence = forms.CharField(widget=forms.TextInput(attrs={'size':100}))
    slot_type = forms.CharField(widget=forms.TextInput(attrs={'size':100}))
