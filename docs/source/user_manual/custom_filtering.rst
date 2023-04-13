Custom Filtering
================

The KBMOD software package provides the ability to conduct further rounds of user-defined filtering using filters derived from the :py:class:`~kbmod.filters.RowFilter` or :py:class:`~kbmod.filters.BatchFilter` base classes. 
RowFilter objects allow you to define a custom function for filtering the data, called ``keep_row`` that takes a single :py:class:`~kbmod.ResultRow` and returns a Boolean indicating whether or not to keep the row. 
BatchFilter objects allow you to define a function for filtering on the entire result set called ``keep_indices`` that takes a :py:class:`~kbmod.ResultList` and returns a list of indices to keep. 
BatchFilters are meant to be used in cases where the filter needs to consider aggregate information about the results, such as using clustering to remove rows, or the user wants to define their own optimizations, such as using a GPU.

Using Row Filters
-----------------

The :py:class:`~kbmod.ResultList` class uses a method ``apply_filter`` to apply the filter to each row in the list of results. 
An optional parameter ``num_threads`` allows the user to run the filter over rows in parallel. 
For example, we could removes rows with fewer than 8 valid observations using::

    my_filter = NumObsFilter(8)
    my_results.apply_filter(my_filter)

All filters include ``get_filter_name()`` function that will return a string representing the filter’s type and configuration. 
For example, the likelihood filter’s name indicates both the minimum and maximum likelihood allowed (where ``None`` indicates no limit): ``LH_Filter_40.0_to_None``.

Using Batch Filters
-------------------

The :py:class:`~kbmod.ResultList` class uses a method ``apply_batch_filter`` to apply the given batch filter. For example, we could cluster rows and remove near duplicates using::

    my_filter = DBSCANFilter("position", 0.025, 100, 100, [0, 50], [0, 1.5], times)
    my_results.apply_batch_filter(my_filter)

All batch filters include ``get_filter_name()`` function that will return a string representing the filter’s type and configuration.


Tracking Filtered Rows
----------------------

In some instances the user might want to track which rows are filtered for retrospective analysis. The :py:class:`~kbmod.ResultList` data structure provides a mechanism to do this via the ``track_filtered = True`` setting. After each round of filtering the filtered rows will be stored in a list accessible by the filter name. Technically the :py:class:`~kbmod.ResultList` maintains a dictionary mapping the filter name to a list of rows removed by that filter. Note that tracking the filtered results will greatly increase the memory usages because filtered tracks are no longer discarded. Therefor, we recommend only using this method for debugging and analysis purposes.

The list of filtered rows can then be accessed using the ``get_filtered()`` function. If a string is passed in to ``get_filtered``, the function will return only those rows removed by the corresponding filter. Otherwise it will return all filtered rows.

User Defined Filters
--------------------

Users can define their own filters by inheriting from the :py:class:`~kbmod.filters.RowFilter` or :py:class:`~kbmod.filters.BatchFilter` base classes and providing an implementation for the ``get_filter_name`` and  ``keep_row`` functions. The ``get_filter_name`` function returns a string with the filter’s type and configuration. 
For row filters, the user defines a ``keep_row`` function that returns a Boolean indicating whether the row is kept (or filtered).
For batch filters, the user defines a ``keep_indices`` that returns a list of indices.
Numerous example filters can be found in the ``src/filters/`` directory.