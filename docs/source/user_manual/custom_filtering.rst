Custom Filtering
================

The KBMOD software package provides the ability to conduct further rounds of user-defined filtering using either the integer indices or a mask. Results are returned in a :py:class:`~kbmod.results.Results` object, which behaves similar to an astropy or pandas table (and is, in fact, backed by an astropy table). Individual results, or rows, can be access by index number::
    
    my_results[0]

and the total number of results can be accessed using the standard ``len()`` function.

The :py:class:`~kbmod.results.Results` object provides ``filter_rows()`` that will both remove the corresponding results from the data and (potentially) track them. The ``filter_rows()`` takes up to two arguments: 1) Information on the rows to filtered and 2) an optional label for tracking the filtered results. The rows to be filtered can be filtered either using a Boolean mask (``True`` means keep and ``False`` means reject) the same size as the data or a list of indices to preserve.

For example the call::
    
    my_results.filter_rows([1, 3, 5], "odds_filter")
    
will preserve only the results in rows 1, 3, and 5. If tracking is enabled for the :py:class:`~kbmod.results.Results` object, it will maintain the filtered rows (everything else) in a separate table under the label "odds_filter". In addition the :py:class:`~kbmod.results.Results` object will always maintain counts of the number of results removed by each filter (label).

Standard Filters
----------------

KBMOD provides a few standard filtering algorithms. See :ref:`Results analysis` for details.

Tracking Filtered Rows
----------------------

In some instances the user might want to track which rows are filtered for retrospective analysis. The :py:class:`~kbmod.ResultList` data structure provides a mechanism to do this via the ``track_filtered = True`` setting. After each round of filtering the filtered rows will be stored in a list accessible by the filter name. Technically the :py:class:`~kbmod.ResultList` maintains a dictionary mapping the filter name to a list of rows removed by that filter. Note that tracking the filtered results will greatly increase the memory usages because filtered tracks are no longer discarded. Therefor, we recommend only using this method for debugging and analysis purposes.

The list of filtered rows can then be accessed using the ``get_filtered()`` function. If a string is passed in to ``get_filtered``, the function will return only those rows removed by the corresponding filter. Otherwise, it will return all filtered rows.

Reverting Filters
-----------------

The :py:class:`~kbmod.results.Results` object provides the option to revert one or all filters. For example if you want to undo the odd-filtering step above, you could apply::

    my_results.revert_filter(label="odds_filter")

which will re-add all the filtered rows with the label "odds_filter". 

Alternatively you can leave the label out to revert all filtered rows. When reverting all filtered rows, it is often beneficial to maintain the information about which row was filtered and why. You can add a new column to the :py:class:`~kbmod.results.Results` object by providing a column name. For example to revert all the filters and include the filter label in a new column called "reason" you could use::

    my_results.revert_filter(add_column="reason")

This is particularly useful when you are expecting a result (e.g. an inserted fake), but do not see it. You can revert all the filters, search for a match in the (expanded) results, and determine why it was filtered. This workflow can help you tune the filtering parameters of different stages.
