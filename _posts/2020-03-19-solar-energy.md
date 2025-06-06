---
keywords: fastai
description: How much energy is actually renewable and humanity's cap in consumption
title: Super simple estimation of available solar energy
toc: true 
badges: true
comments: true
categories: [energy]
nb_path: _notebooks/2020-03-19-solar-energy.ipynb
layout: notebook
---

<!--
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: _notebooks/2020-03-19-solar-energy.ipynb
-->

<div class="container" id="notebook-container">
        
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Solar-energy">Solar energy<a class="anchor-link" href="#Solar-energy"> </a></h1><h2 id="Stefan-boltzmann's-law">Stefan boltzmann's law<a class="anchor-link" href="#Stefan-boltzmann's-law"> </a></h2><p>$ \text{Surface energy} = \sigma T^4$</p>
<p>For the sun, $T = \text{5,778 }K$</p>
<p>$\sigma = 5.67 \times 10 ^{-8} W.m^{-2}.K^{-4}$</p>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sympy.physics.units</span> <span class="kn">import</span> <span class="n">K</span><span class="p">,</span> <span class="n">W</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">giga</span>

<span class="n">sigma</span> <span class="o">=</span> <span class="mf">5.67</span> <span class="o">*</span> <span class="mi">10</span><span class="o">**</span><span class="p">(</span><span class="o">-</span><span class="mi">8</span><span class="p">)</span> <span class="o">*</span> <span class="n">W</span> <span class="o">*</span><span class="n">m</span><span class="o">**</span><span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="p">)</span> <span class="o">*</span> <span class="n">K</span><span class="o">**</span><span class="p">(</span><span class="o">-</span><span class="mi">4</span><span class="p">)</span>
<span class="n">T</span> <span class="o">=</span> <span class="mi">5778</span> <span class="o">*</span> <span class="n">K</span>
<span class="n">surface_energy</span> <span class="o">=</span> <span class="n">sigma</span> <span class="o">*</span> <span class="n">T</span><span class="o">**</span><span class="mi">4</span>
<span class="nb">print</span><span class="p">(</span><span class="n">surface_energy</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>63196526.5460292*watt/meter**2
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Total-emitted-solar-energy">Total emitted solar energy<a class="anchor-link" href="#Total-emitted-solar-energy"> </a></h2><p>$ Radiation = \text{Surface of the sun} \times \text{Surface energy} $</p>
<p>$ Radiation = 4 \pi r^2 \times \text{Surface energy} $</p>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sympy</span> <span class="kn">import</span> <span class="o">*</span>

<span class="n">r_sun</span> <span class="o">=</span> <span class="mi">696_340</span> <span class="o">*</span> <span class="mi">1000</span> <span class="o">*</span><span class="n">m</span>
<span class="n">surface_of_sun</span> <span class="o">=</span> <span class="mi">4</span> <span class="o">*</span> <span class="n">pi</span> <span class="o">*</span> <span class="n">r_sun</span> <span class="o">**</span> <span class="mi">2</span> 
<span class="n">radiation</span> <span class="o">=</span> <span class="n">surface_of_sun</span> <span class="o">*</span> <span class="n">surface_energy</span>

<span class="nb">print</span><span class="p">(</span><span class="n">radiation</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>1.22573302243694e+26*pi*watt
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Energy-received-at-earth-average-distance">Energy received at earth average distance<a class="anchor-link" href="#Energy-received-at-earth-average-distance"> </a></h2><p>$ \text{Radiation received} = \frac{\text{Total sun radiation}}{ \text{sphere at earth's distance}}$</p>
<p>$ \text{Radiation received} = \frac{Radiation}{ 4 \pi D_{earth-sun}^2} $</p>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">R_earth</span> <span class="o">=</span> <span class="mi">6_371</span> <span class="o">*</span> <span class="mi">1000</span> <span class="o">*</span> <span class="n">m</span>
<span class="n">D_earth_sun</span> <span class="o">=</span> <span class="mf">148.88</span> <span class="o">*</span> <span class="mi">10</span><span class="o">**</span><span class="mi">6</span> <span class="o">*</span> <span class="mi">1000</span> <span class="o">*</span> <span class="n">m</span>
<span class="n">earth_perp_surface</span> <span class="o">=</span> <span class="n">pi</span> <span class="o">*</span> <span class="n">R_earth</span> <span class="o">**</span><span class="mi">2</span>
<span class="n">sphere</span> <span class="o">=</span> <span class="mi">4</span> <span class="o">*</span> <span class="n">pi</span> <span class="o">*</span> <span class="n">D_earth_sun</span> <span class="o">**</span><span class="mi">2</span>

<span class="n">radiation_received</span> <span class="o">=</span> <span class="n">radiation</span> <span class="o">/</span> <span class="n">sphere</span>
<span class="nb">print</span><span class="p">(</span><span class="n">radiation_received</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>1382.49374484614*watt/meter**2
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Energy-received-by-the-earth-surface-(before-atmosphere)">Energy received by the earth surface (before atmosphere)<a class="anchor-link" href="#Energy-received-by-the-earth-surface-(before-atmosphere)"> </a></h2><p>$ \text{Energy received} = \text{radiation received} \times \frac{ \text{visible surface}}{ \text{earth's surface}} $</p>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">power_received</span> <span class="o">=</span> <span class="n">radiation_received</span> <span class="o">*</span> <span class="n">pi</span> <span class="o">*</span> <span class="n">R_earth</span> <span class="o">**</span><span class="mi">2</span>
<span class="n">surface_power_received</span> <span class="o">=</span> <span class="n">power_received</span> <span class="o">/</span> <span class="p">(</span><span class="mi">4</span> <span class="o">*</span> <span class="n">pi</span> <span class="o">*</span> <span class="n">R_earth</span> <span class="o">**</span><span class="mi">2</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">surface_power_received</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">power_received</span><span class="o">.</span><span class="n">n</span><span class="p">())</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>345.623436211536*watt/meter**2
1.76290235470883e+17*watt
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<blockquote><p>RADIATION RECEIVED BY SYSTEM EARTH  = $345 W.m^{-2}$</p>
<p>MAXIMUM POWER WITH EARTH "DYSON SPHERE": $176 PW$</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Human-consumption">Human consumption<a class="anchor-link" href="#Human-consumption"> </a></h1><p>13 511 MTep <a href="https://www.iea.org/data-and-statistics?country=WORLD&amp;fuel=Energy%20supply&amp;indicator=Total%20primary%20energy%20supply%20%28TPES%29%20by%20source">Source International Energy agency</a></p>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sympy.physics.units</span> <span class="kn">import</span> <span class="n">J</span><span class="p">,</span> <span class="n">s</span><span class="p">,</span> <span class="n">W</span>
<span class="kn">from</span> <span class="nn">sympy.physics.units.util</span> <span class="kn">import</span> <span class="n">convert_to</span>

<span class="n">million</span> <span class="o">=</span> <span class="mi">10</span> <span class="o">**</span><span class="mi">6</span>
<span class="n">kilo</span> <span class="o">=</span> <span class="mi">10</span><span class="o">**</span><span class="mi">3</span>
<span class="n">giga</span> <span class="o">=</span> <span class="mi">10</span> <span class="o">**</span> <span class="mi">9</span>
<span class="n">toe</span> <span class="o">=</span> <span class="mf">41.868</span> <span class="o">*</span> <span class="n">giga</span> <span class="o">*</span> <span class="n">J</span>
<span class="n">ktoe</span> <span class="o">=</span> <span class="n">kilo</span> <span class="o">*</span> <span class="n">toe</span>
<span class="n">Mtoe</span> <span class="o">=</span> <span class="n">million</span> <span class="o">*</span> <span class="n">toe</span>

<span class="n">hour</span> <span class="o">=</span> <span class="mi">60</span> <span class="o">*</span> <span class="mi">60</span> <span class="o">*</span> <span class="n">s</span>
<span class="n">year</span> <span class="o">=</span> <span class="mi">24</span> <span class="o">*</span> <span class="n">h</span> <span class="o">*</span> <span class="mf">365.25</span>

<span class="n">base</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">([</span><span class="mi">3852538</span><span class="p">,</span><span class="mi">2949909</span><span class="p">,</span><span class="mi">670298</span><span class="p">,</span><span class="mi">335519</span><span class="p">,</span><span class="mi">204190</span><span class="p">,</span><span class="mi">1286064</span><span class="p">,</span><span class="mi">4329220</span><span class="p">])</span>
<span class="n">Humanity_total_annual_consumption</span> <span class="o">=</span> <span class="n">base</span> <span class="o">*</span> <span class="n">ktoe</span>


<span class="n">humanity_power_consumption</span> <span class="o">=</span> <span class="n">Humanity_total_annual_consumption</span> <span class="o">/</span> <span class="n">year</span>
<span class="nb">print</span><span class="p">(</span><span class="n">convert_to</span><span class="p">(</span><span class="n">humanity_power_consumption</span><span class="o">.</span><span class="n">n</span><span class="p">(),</span> <span class="p">[</span><span class="n">W</span><span class="p">])</span><span class="o">.</span><span class="n">n</span><span class="p">())</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>18080149776408.9*watt
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">convert_to</span><span class="p">(</span><span class="n">humanity_power_consumption</span> <span class="o">/</span> <span class="n">power_received</span> <span class="o">*</span> <span class="mi">100</span><span class="p">,</span> <span class="p">[</span><span class="n">J</span><span class="p">,</span> <span class="n">s</span><span class="p">])</span><span class="o">.</span><span class="n">n</span><span class="p">())</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>0.0102558997258785
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We are currently consuming <strong>0.01% of the maximum capacity of the earth covered by a Dyson sphere of solar panels</strong>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="A-bit-more-realistic-approach">A bit more realistic approach<a class="anchor-link" href="#A-bit-more-realistic-approach"> </a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>After the atmosphere only $168 W.m^{-2}$ hit the surface. It's quite complicated to infer it depends on the wavelength of the incoming light, clouds, composition of the atmosphere and so on, so we just take the value from <a href="https://fr.wikipedia.org/wiki/Bilan_radiatif_de_la_Terre">here</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Then we only have 29% of the earth surface that is landmass (where we can reasonably put solar panels in large quantity)</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Of that 31% is covered in forest which are already some natural solar panels we don't  want to remove (for other obvious reasons) <a href="http://www.earth-policy.org/indicators/C56/forests_2012">source</a>
And 38.4% is covered of agricultural land <a href="https://en.wikipedia.org/wiki/Agricultural_land">source</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Then solar panels are not 100% efficient. They are roughly only 20% efficient with current technology at a reasonable cost.</p>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">earth_power_received</span> <span class="o">=</span> <span class="mi">168</span> <span class="o">*</span> <span class="n">W</span> <span class="o">*</span> <span class="n">m</span> <span class="o">**</span><span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="p">)</span>

<span class="n">available_surface</span> <span class="o">=</span> <span class="mi">4</span> <span class="o">*</span> <span class="n">pi</span> <span class="o">*</span> <span class="n">R_earth</span> <span class="o">**</span><span class="mi">2</span> <span class="o">*</span> <span class="mf">0.29</span> <span class="o">*</span> <span class="p">(</span><span class="mi">1</span> <span class="o">-.</span><span class="mi">31</span> <span class="o">-</span> <span class="o">.</span><span class="mi">384</span><span class="p">)</span>

<span class="n">max_power</span> <span class="o">=</span> <span class="n">earth_power_received</span> <span class="o">*</span> <span class="n">available_surface</span> <span class="o">*</span> <span class="mf">0.2</span>

<span class="nb">print</span><span class="p">(</span><span class="n">max_power</span><span class="o">.</span><span class="n">n</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="n">convert_to</span><span class="p">(</span><span class="n">humanity_power_consumption</span> <span class="o">/</span> <span class="n">max_power</span> <span class="o">*</span><span class="mi">100</span><span class="p">,</span> <span class="p">[</span><span class="n">J</span><span class="p">,</span> <span class="n">s</span><span class="p">])</span><span class="o">.</span><span class="n">n</span><span class="p">())</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>1.52084087357243e+15*watt
1.18882587196246
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion"> </a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In the end we are currently consuming <strong>1.2% of the realistic available solar power energy</strong>. That's would require posing solar panels everywhere on the planet that is not a forest or agricultural land. And we don't account yet for Energy return on energy invested (EROEI) which is likely to increase that percentage.</p>
<p>NB: This is a very superficial attempt to evaluate these numbers, however the result should be correct within an order of magnitude.</p>

</div>
</div>
</div>
</div>
 

