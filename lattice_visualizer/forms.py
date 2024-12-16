from django import forms


class BasisInputForm(forms.Form):
    # Matrix 1 fields as Integer Fields
    matrix1_00 = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=3)
    matrix1_01 = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=1)
    matrix1_10 = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=1)
    matrix1_11 = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=3)

    # Optional Matrix 2 fields
    matrix2_00 = forms.IntegerField(required=False,
                                    widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=4)
    matrix2_01 = forms.IntegerField(required=False,
                                    widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=4)
    matrix2_10 = forms.IntegerField(required=False,
                                    widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=7)
    matrix2_11 = forms.IntegerField(required=False,
                                    widget=forms.TextInput(attrs={'class': 'matrix-input', 'pattern': r'-?\d+'}), initial=5)
