---
layout: schedule
permalink: /EAI-fall-2025/lectures/
title: Schedule
---

{% assign current_module = 0 %}
{% assign skip_classes = 0 %}
{% assign prev_date = 0 %}

{% for item in site.data.lectures %}
{% if item.date %}
{% assign lecture = item %}
{% assign event_type = "upcoming" %}
{% assign today_date = "now" | date: "%s" | divided_by: 86400 %}
{% assign lecture_date = lecture.date | date: "%s" | divided_by: 86400 %}
{% if today_date > lecture_date %}
    {% assign event_type = "past" %}
{% elsif today_date <= lecture_date and today_date > prev_date %}
    {% assign event_type = "warning" %}
{% endif %}
{% assign prev_date = lecture_date %}

<tr class="{{ event_type }}">
    <th scope="row">{{ lecture.date }}</th>
    {% if lecture.title contains 'No class' or forloop.last %}
    {% assign skip_classes = skip_classes | plus: 1 %}
    <td colspan="4" align="center">{{ lecture.title }}</td>
    {% else %}
    <td>
        Lecture #{{ forloop.index | minus: current_module | minus: skip_classes }}:
        <br />
        {{ lecture.title }}
    </td>
    <td>
        {% if lecture.slides %}
          <a href="{{ lecture.slides }}" target="_blank">slides</a>
        {% else %}
          slides
        {% endif %}
        {% if lecture.annotated %}
          (<a href="{{ lecture.annotated }}" target="_blank">annotated</a>)
        {% endif %}
    </td>
    <td>
        <p>{{ lecture.logistics }}</p>
    </td>
    {% endif %}
</tr>
{% else %}
{% assign current_module = current_module | plus: 1 %}
{% assign module = item %}
<tr class="info">
    <td colspan="4" align="center"><strong>{{ module.title }}</strong></td>
</tr>
{% endif %}
{% endfor %}
