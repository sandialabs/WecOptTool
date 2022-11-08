..
   Adapted from https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
   See: https://github.com/sphinx-doc/sphinx/issues/7912

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :special-members: __init__,


{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}
.. autosummary::
    :toctree:
{% for item in attributes %}
    {{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}
.. autosummary::
    :toctree:
    :nosignatures:
{% for item in methods %}
    {%- if not item.startswith('_') %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}