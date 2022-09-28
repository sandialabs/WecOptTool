..
   Adapted from https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
   See: https://github.com/sphinx-doc/sphinx/issues/7912

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
