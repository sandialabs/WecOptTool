..
    Adapted from https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
    See: https://github.com/sphinx-doc/sphinx/issues/7912

{{ 'API Documentation' | escape | underline}}

.. automodule:: {{ fullname }}

{% block modules %}
{% if modules %}
.. rubric:: {{ _('Modules') }}
.. autosummary::
   :template: module.rst
   :toctree:
{% for item in modules %}
{% if item != 'wecopttool.core' %}
   {{ item }}
{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: {{ _('Classes') }}
.. autosummary::
    :toctree:
    :template: class.rst
    :nosignatures:
{% for item in classes %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

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
